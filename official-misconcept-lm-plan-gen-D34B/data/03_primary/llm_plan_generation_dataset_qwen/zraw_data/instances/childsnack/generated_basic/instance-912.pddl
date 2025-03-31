; child-snack task with 3 children and 0.31 gluten factor 
; constant factor of 1.5
; random seed: 906452

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child349 child361 child221 - child
    bread68 bread472 bread141 - bread-portion
    content301 content124 content9 - content-portion
    tray138 - tray
    table240 table343 table25 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 - sandwich
  )
  (:init
     (at tray138 kitchen)
     (at_kitchen_bread bread68)
     (at_kitchen_bread bread472)
     (at_kitchen_bread bread141)
     (at_kitchen_content content301)
     (at_kitchen_content content124)
     (at_kitchen_content content9)
     (not_allergic_gluten child221)
     (not_allergic_gluten child349)
     (not_allergic_gluten child361)
     (waiting child349 table25)
     (waiting child361 table25)
     (waiting child221 table240)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
  )
  (:goal
    (and
     (served child349)
     (served child361)
     (served child221)
    )
  )
)
