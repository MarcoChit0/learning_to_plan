; child-snack task with 2 children and 0.39 gluten factor 
; constant factor of 1.9
; random seed: 819376

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child158 child271 - child
    bread307 bread311 - bread-portion
    content476 content433 - content-portion
    tray67 tray29 - tray
    table321 table412 table170 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray67 kitchen)
     (at tray29 kitchen)
     (at_kitchen_bread bread307)
     (at_kitchen_bread bread311)
     (at_kitchen_content content476)
     (at_kitchen_content content433)
     (not_allergic_gluten child271)
     (not_allergic_gluten child158)
     (waiting child158 table170)
     (waiting child271 table412)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child158)
     (served child271)
    )
  )
)
