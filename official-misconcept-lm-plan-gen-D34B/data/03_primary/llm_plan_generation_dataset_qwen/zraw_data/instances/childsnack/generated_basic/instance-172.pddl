; child-snack task with 2 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 422409

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child55 child450 - child
    bread21 bread199 - bread-portion
    content346 content73 - content-portion
    tray84 tray382 tray79 tray184 - tray
    table65 table51 table374 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray84 kitchen)
     (at tray382 kitchen)
     (at tray79 kitchen)
     (at tray184 kitchen)
     (at_kitchen_bread bread21)
     (at_kitchen_bread bread199)
     (at_kitchen_content content346)
     (at_kitchen_content content73)
     (not_allergic_gluten child55)
     (not_allergic_gluten child450)
     (waiting child55 table374)
     (waiting child450 table374)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child55)
     (served child450)
    )
  )
)
