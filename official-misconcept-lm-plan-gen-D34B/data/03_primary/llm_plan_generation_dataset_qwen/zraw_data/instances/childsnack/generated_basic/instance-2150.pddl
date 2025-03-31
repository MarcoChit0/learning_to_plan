; child-snack task with 2 children and 0.53 gluten factor 
; constant factor of 1.9
; random seed: 809428

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child275 child46 - child
    bread343 bread202 - bread-portion
    content463 content45 - content-portion
    tray21 tray481 tray304 - tray
    table169 table28 table5 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray21 kitchen)
     (at tray481 kitchen)
     (at tray304 kitchen)
     (at_kitchen_bread bread343)
     (at_kitchen_bread bread202)
     (at_kitchen_content content463)
     (at_kitchen_content content45)
     (no_gluten_bread bread202)
     (no_gluten_content content45)
     (allergic_gluten child46)
     (not_allergic_gluten child275)
     (waiting child275 table5)
     (waiting child46 table28)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child275)
     (served child46)
    )
  )
)
