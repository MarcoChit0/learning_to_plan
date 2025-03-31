; child-snack task with 3 children and 0.31 gluten factor 
; constant factor of 1.5
; random seed: 341067

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child389 child380 child49 - child
    bread117 bread366 bread227 - bread-portion
    content199 content394 content242 - content-portion
    tray12 - tray
    table38 table87 table76 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 - sandwich
  )
  (:init
     (at tray12 kitchen)
     (at_kitchen_bread bread117)
     (at_kitchen_bread bread366)
     (at_kitchen_bread bread227)
     (at_kitchen_content content199)
     (at_kitchen_content content394)
     (at_kitchen_content content242)
     (not_allergic_gluten child389)
     (not_allergic_gluten child380)
     (not_allergic_gluten child49)
     (waiting child389 table87)
     (waiting child380 table76)
     (waiting child49 table38)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
  )
  (:goal
    (and
     (served child389)
     (served child380)
     (served child49)
    )
  )
)
