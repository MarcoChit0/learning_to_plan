; child-snack task with 1 children and 0.77 gluten factor 
; constant factor of 1.9
; random seed: 936839

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child133 - child
    bread297 - bread-portion
    content220 - content-portion
    tray352 tray497 tray253 tray139 - tray
    table366 table435 table100 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray352 kitchen)
     (at tray497 kitchen)
     (at tray253 kitchen)
     (at tray139 kitchen)
     (at_kitchen_bread bread297)
     (at_kitchen_content content220)
     (not_allergic_gluten child133)
     (waiting child133 table435)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child133)
    )
  )
)
