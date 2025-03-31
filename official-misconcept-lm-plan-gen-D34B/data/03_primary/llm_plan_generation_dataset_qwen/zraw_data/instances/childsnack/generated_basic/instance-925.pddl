; child-snack task with 3 children and 0.31 gluten factor 
; constant factor of 1.5
; random seed: 147197

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child244 child70 child271 - child
    bread335 bread111 bread77 - bread-portion
    content321 content259 content276 - content-portion
    tray59 - tray
    table94 table72 table97 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 - sandwich
  )
  (:init
     (at tray59 kitchen)
     (at_kitchen_bread bread335)
     (at_kitchen_bread bread111)
     (at_kitchen_bread bread77)
     (at_kitchen_content content321)
     (at_kitchen_content content259)
     (at_kitchen_content content276)
     (not_allergic_gluten child271)
     (not_allergic_gluten child244)
     (not_allergic_gluten child70)
     (waiting child244 table94)
     (waiting child70 table72)
     (waiting child271 table72)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
  )
  (:goal
    (and
     (served child244)
     (served child70)
     (served child271)
    )
  )
)
