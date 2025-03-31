; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.9
; random seed: 124086

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child499 child223 - child
    bread202 bread476 - bread-portion
    content1 content449 - content-portion
    tray227 tray365 - tray
    table415 table50 table443 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray227 kitchen)
     (at tray365 kitchen)
     (at_kitchen_bread bread202)
     (at_kitchen_bread bread476)
     (at_kitchen_content content1)
     (at_kitchen_content content449)
     (not_allergic_gluten child499)
     (not_allergic_gluten child223)
     (waiting child499 table50)
     (waiting child223 table50)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child499)
     (served child223)
    )
  )
)
