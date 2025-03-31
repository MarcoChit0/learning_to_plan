; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.3
; random seed: 424138

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child430 child439 - child
    bread389 bread316 - bread-portion
    content355 content465 - content-portion
    tray400 tray295 tray150 - tray
    table441 table42 table12 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray400 kitchen)
     (at tray295 kitchen)
     (at tray150 kitchen)
     (at_kitchen_bread bread389)
     (at_kitchen_bread bread316)
     (at_kitchen_content content355)
     (at_kitchen_content content465)
     (not_allergic_gluten child430)
     (not_allergic_gluten child439)
     (waiting child430 table12)
     (waiting child439 table441)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child430)
     (served child439)
    )
  )
)
