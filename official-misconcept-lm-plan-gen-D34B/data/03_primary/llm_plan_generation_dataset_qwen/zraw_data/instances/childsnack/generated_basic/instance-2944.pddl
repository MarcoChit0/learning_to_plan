; child-snack task with 1 children and 0.69 gluten factor 
; constant factor of 1.3
; random seed: 227013

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child77 - child
    bread125 - bread-portion
    content2 - content-portion
    tray281 tray391 tray453 tray91 - tray
    table380 table384 table383 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray281 kitchen)
     (at tray391 kitchen)
     (at tray453 kitchen)
     (at tray91 kitchen)
     (at_kitchen_bread bread125)
     (at_kitchen_content content2)
     (not_allergic_gluten child77)
     (waiting child77 table380)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child77)
    )
  )
)
