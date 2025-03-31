; child-snack task with 2 children and 0.42 gluten factor 
; constant factor of 1.3
; random seed: 527330

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child411 child451 - child
    bread40 bread33 - bread-portion
    content348 content472 - content-portion
    tray355 tray77 - tray
    table348 table124 table132 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray355 kitchen)
     (at tray77 kitchen)
     (at_kitchen_bread bread40)
     (at_kitchen_bread bread33)
     (at_kitchen_content content348)
     (at_kitchen_content content472)
     (not_allergic_gluten child411)
     (not_allergic_gluten child451)
     (waiting child411 table132)
     (waiting child451 table132)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child411)
     (served child451)
    )
  )
)
