; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.5
; random seed: 374531

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child442 - child
    bread269 - bread-portion
    content14 - content-portion
    tray104 tray370 tray482 tray418 - tray
    table298 table339 table334 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray104 kitchen)
     (at tray370 kitchen)
     (at tray482 kitchen)
     (at tray418 kitchen)
     (at_kitchen_bread bread269)
     (at_kitchen_content content14)
     (not_allergic_gluten child442)
     (waiting child442 table339)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child442)
    )
  )
)
