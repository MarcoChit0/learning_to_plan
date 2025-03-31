; child-snack task with 3 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 916883

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child136 child369 child368 - child
    bread145 bread337 bread424 - bread-portion
    content48 content451 content255 - content-portion
    tray67 tray402 tray475 tray299 - tray
    table263 table350 table425 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray67 kitchen)
     (at tray402 kitchen)
     (at tray475 kitchen)
     (at tray299 kitchen)
     (at_kitchen_bread bread145)
     (at_kitchen_bread bread337)
     (at_kitchen_bread bread424)
     (at_kitchen_content content48)
     (at_kitchen_content content451)
     (at_kitchen_content content255)
     (not_allergic_gluten child136)
     (not_allergic_gluten child369)
     (not_allergic_gluten child368)
     (waiting child136 table425)
     (waiting child369 table263)
     (waiting child368 table263)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child136)
     (served child369)
     (served child368)
    )
  )
)
