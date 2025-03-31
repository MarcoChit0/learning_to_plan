; child-snack task with 2 children and 0.77 gluten factor 
; constant factor of 1.3
; random seed: 414596

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child308 child241 - child
    bread55 bread121 - bread-portion
    content149 content498 - content-portion
    tray177 - tray
    table203 table444 table86 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray177 kitchen)
     (at_kitchen_bread bread55)
     (at_kitchen_bread bread121)
     (at_kitchen_content content149)
     (at_kitchen_content content498)
     (no_gluten_bread bread121)
     (no_gluten_content content149)
     (allergic_gluten child308)
     (not_allergic_gluten child241)
     (waiting child308 table203)
     (waiting child241 table444)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child308)
     (served child241)
    )
  )
)
