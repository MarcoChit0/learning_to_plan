; child-snack task with 3 children and 0.31 gluten factor 
; constant factor of 1.5
; random seed: 24985

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child91 child270 child175 - child
    bread35 bread262 bread12 - bread-portion
    content307 content352 content198 - content-portion
    tray243 - tray
    table206 table291 table392 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 - sandwich
  )
  (:init
     (at tray243 kitchen)
     (at_kitchen_bread bread35)
     (at_kitchen_bread bread262)
     (at_kitchen_bread bread12)
     (at_kitchen_content content307)
     (at_kitchen_content content352)
     (at_kitchen_content content198)
     (not_allergic_gluten child91)
     (not_allergic_gluten child270)
     (not_allergic_gluten child175)
     (waiting child91 table206)
     (waiting child270 table291)
     (waiting child175 table392)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
  )
  (:goal
    (and
     (served child91)
     (served child270)
     (served child175)
    )
  )
)
