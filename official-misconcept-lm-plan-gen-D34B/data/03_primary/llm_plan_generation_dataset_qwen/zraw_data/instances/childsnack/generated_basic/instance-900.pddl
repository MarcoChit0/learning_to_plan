; child-snack task with 3 children and 0.31 gluten factor 
; constant factor of 1.5
; random seed: 418060

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child388 child159 child286 - child
    bread193 bread141 bread442 - bread-portion
    content263 content142 content498 - content-portion
    tray26 - tray
    table385 table221 table495 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 - sandwich
  )
  (:init
     (at tray26 kitchen)
     (at_kitchen_bread bread193)
     (at_kitchen_bread bread141)
     (at_kitchen_bread bread442)
     (at_kitchen_content content263)
     (at_kitchen_content content142)
     (at_kitchen_content content498)
     (not_allergic_gluten child388)
     (not_allergic_gluten child286)
     (not_allergic_gluten child159)
     (waiting child388 table221)
     (waiting child159 table385)
     (waiting child286 table221)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
  )
  (:goal
    (and
     (served child388)
     (served child159)
     (served child286)
    )
  )
)
