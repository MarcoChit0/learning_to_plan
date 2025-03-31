; child-snack task with 3 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 948410

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child474 child320 child199 - child
    bread355 bread75 bread488 - bread-portion
    content210 content149 content212 - content-portion
    tray198 - tray
    table118 table436 table361 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray198 kitchen)
     (at_kitchen_bread bread355)
     (at_kitchen_bread bread75)
     (at_kitchen_bread bread488)
     (at_kitchen_content content210)
     (at_kitchen_content content149)
     (at_kitchen_content content212)
     (not_allergic_gluten child474)
     (not_allergic_gluten child320)
     (not_allergic_gluten child199)
     (waiting child474 table118)
     (waiting child320 table118)
     (waiting child199 table436)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child474)
     (served child320)
     (served child199)
    )
  )
)
