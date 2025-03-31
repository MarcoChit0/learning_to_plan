; child-snack task with 2 children and 0.77 gluten factor 
; constant factor of 1.3
; random seed: 817035

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child184 child178 - child
    bread287 bread17 - bread-portion
    content108 content274 - content-portion
    tray143 - tray
    table105 table252 table191 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray143 kitchen)
     (at_kitchen_bread bread287)
     (at_kitchen_bread bread17)
     (at_kitchen_content content108)
     (at_kitchen_content content274)
     (no_gluten_bread bread17)
     (no_gluten_content content108)
     (allergic_gluten child184)
     (not_allergic_gluten child178)
     (waiting child184 table252)
     (waiting child178 table105)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child184)
     (served child178)
    )
  )
)
