; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.9
; random seed: 1279

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child329 child281 - child
    bread418 bread189 - bread-portion
    content107 content292 - content-portion
    tray310 tray338 tray463 tray186 - tray
    table101 table44 table316 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray310 kitchen)
     (at tray338 kitchen)
     (at tray463 kitchen)
     (at tray186 kitchen)
     (at_kitchen_bread bread418)
     (at_kitchen_bread bread189)
     (at_kitchen_content content107)
     (at_kitchen_content content292)
     (not_allergic_gluten child329)
     (not_allergic_gluten child281)
     (waiting child329 table316)
     (waiting child281 table44)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child329)
     (served child281)
    )
  )
)
