; child-snack task with 3 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 926646

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child345 child402 child91 - child
    bread16 bread181 bread366 - bread-portion
    content460 content134 content217 - content-portion
    tray218 - tray
    table222 table288 table207 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray218 kitchen)
     (at_kitchen_bread bread16)
     (at_kitchen_bread bread181)
     (at_kitchen_bread bread366)
     (at_kitchen_content content460)
     (at_kitchen_content content134)
     (at_kitchen_content content217)
     (not_allergic_gluten child402)
     (not_allergic_gluten child91)
     (not_allergic_gluten child345)
     (waiting child345 table207)
     (waiting child402 table288)
     (waiting child91 table288)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child345)
     (served child402)
     (served child91)
    )
  )
)
