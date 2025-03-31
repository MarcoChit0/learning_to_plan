; child-snack task with 1 children and 0.48 gluten factor 
; constant factor of 1.7
; random seed: 680764

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child414 - child
    bread385 - bread-portion
    content292 - content-portion
    tray255 tray455 tray151 - tray
    table141 table380 table307 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray255 kitchen)
     (at tray455 kitchen)
     (at tray151 kitchen)
     (at_kitchen_bread bread385)
     (at_kitchen_content content292)
     (not_allergic_gluten child414)
     (waiting child414 table307)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child414)
    )
  )
)
