; child-snack task with 2 children and 0.36 gluten factor 
; constant factor of 1.9
; random seed: 77008

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child87 child294 - child
    bread230 bread224 - bread-portion
    content229 content181 - content-portion
    tray43 tray499 - tray
    table185 table141 table284 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray43 kitchen)
     (at tray499 kitchen)
     (at_kitchen_bread bread230)
     (at_kitchen_bread bread224)
     (at_kitchen_content content229)
     (at_kitchen_content content181)
     (not_allergic_gluten child87)
     (not_allergic_gluten child294)
     (waiting child87 table284)
     (waiting child294 table141)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child87)
     (served child294)
    )
  )
)
