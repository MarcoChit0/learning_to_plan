; child-snack task with 2 children and 0.39 gluten factor 
; constant factor of 1.9
; random seed: 344671

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child155 child169 - child
    bread187 bread34 - bread-portion
    content359 content235 - content-portion
    tray205 tray240 - tray
    table96 table205 table458 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray205 kitchen)
     (at tray240 kitchen)
     (at_kitchen_bread bread187)
     (at_kitchen_bread bread34)
     (at_kitchen_content content359)
     (at_kitchen_content content235)
     (not_allergic_gluten child169)
     (not_allergic_gluten child155)
     (waiting child155 table96)
     (waiting child169 table96)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child155)
     (served child169)
    )
  )
)
