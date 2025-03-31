; child-snack task with 2 children and 0.8 gluten factor 
; constant factor of 1.7
; random seed: 498793

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child351 child400 - child
    bread461 bread493 - bread-portion
    content225 content193 - content-portion
    tray93 tray33 - tray
    table45 table276 table359 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray93 kitchen)
     (at tray33 kitchen)
     (at_kitchen_bread bread461)
     (at_kitchen_bread bread493)
     (at_kitchen_content content225)
     (at_kitchen_content content193)
     (no_gluten_bread bread493)
     (no_gluten_content content225)
     (allergic_gluten child400)
     (not_allergic_gluten child351)
     (waiting child351 table45)
     (waiting child400 table276)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child351)
     (served child400)
    )
  )
)
