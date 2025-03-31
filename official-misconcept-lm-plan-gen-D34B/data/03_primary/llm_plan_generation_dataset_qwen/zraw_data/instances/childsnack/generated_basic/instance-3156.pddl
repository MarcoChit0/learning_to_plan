; child-snack task with 2 children and 0.48 gluten factor 
; constant factor of 1.9
; random seed: 366516

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child369 child5 - child
    bread7 bread87 - bread-portion
    content187 content65 - content-portion
    tray30 tray354 - tray
    table32 table460 table89 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray30 kitchen)
     (at tray354 kitchen)
     (at_kitchen_bread bread7)
     (at_kitchen_bread bread87)
     (at_kitchen_content content187)
     (at_kitchen_content content65)
     (not_allergic_gluten child5)
     (not_allergic_gluten child369)
     (waiting child369 table460)
     (waiting child5 table32)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child369)
     (served child5)
    )
  )
)
