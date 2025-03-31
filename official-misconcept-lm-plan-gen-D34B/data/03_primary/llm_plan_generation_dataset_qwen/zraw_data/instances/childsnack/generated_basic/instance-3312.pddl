; child-snack task with 2 children and 0.49 gluten factor 
; constant factor of 1.3
; random seed: 630255

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child245 child65 - child
    bread1 bread341 - bread-portion
    content200 content113 - content-portion
    tray243 tray320 - tray
    table416 table366 table242 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray243 kitchen)
     (at tray320 kitchen)
     (at_kitchen_bread bread1)
     (at_kitchen_bread bread341)
     (at_kitchen_content content200)
     (at_kitchen_content content113)
     (not_allergic_gluten child245)
     (not_allergic_gluten child65)
     (waiting child245 table416)
     (waiting child65 table366)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child245)
     (served child65)
    )
  )
)
