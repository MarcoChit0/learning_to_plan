; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.3
; random seed: 616506

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child63 - child
    bread135 - bread-portion
    content178 - content-portion
    tray164 tray271 tray426 - tray
    table199 table477 table371 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray164 kitchen)
     (at tray271 kitchen)
     (at tray426 kitchen)
     (at_kitchen_bread bread135)
     (at_kitchen_content content178)
     (not_allergic_gluten child63)
     (waiting child63 table477)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child63)
    )
  )
)
