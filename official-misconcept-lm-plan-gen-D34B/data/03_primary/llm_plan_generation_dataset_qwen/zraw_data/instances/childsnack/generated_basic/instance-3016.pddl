; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.9
; random seed: 319506

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child255 child52 - child
    bread47 bread416 - bread-portion
    content390 content59 - content-portion
    tray76 tray345 tray378 tray123 - tray
    table231 table85 table244 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray76 kitchen)
     (at tray345 kitchen)
     (at tray378 kitchen)
     (at tray123 kitchen)
     (at_kitchen_bread bread47)
     (at_kitchen_bread bread416)
     (at_kitchen_content content390)
     (at_kitchen_content content59)
     (not_allergic_gluten child255)
     (not_allergic_gluten child52)
     (waiting child255 table244)
     (waiting child52 table231)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child255)
     (served child52)
    )
  )
)
