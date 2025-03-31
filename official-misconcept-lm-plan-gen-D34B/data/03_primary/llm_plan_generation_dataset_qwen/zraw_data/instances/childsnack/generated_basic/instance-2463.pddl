; child-snack task with 2 children and 0.39 gluten factor 
; constant factor of 1.9
; random seed: 196249

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child318 child392 - child
    bread87 bread107 - bread-portion
    content212 content463 - content-portion
    tray345 tray99 - tray
    table288 table436 table338 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray345 kitchen)
     (at tray99 kitchen)
     (at_kitchen_bread bread87)
     (at_kitchen_bread bread107)
     (at_kitchen_content content212)
     (at_kitchen_content content463)
     (not_allergic_gluten child318)
     (not_allergic_gluten child392)
     (waiting child318 table288)
     (waiting child392 table436)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child318)
     (served child392)
    )
  )
)
