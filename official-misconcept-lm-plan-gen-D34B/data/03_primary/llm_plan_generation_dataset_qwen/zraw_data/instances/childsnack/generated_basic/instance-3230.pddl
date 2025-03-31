; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.9
; random seed: 792878

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child112 child175 - child
    bread200 bread87 - bread-portion
    content328 content19 - content-portion
    tray52 tray2 tray294 tray104 - tray
    table27 table246 table395 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray52 kitchen)
     (at tray2 kitchen)
     (at tray294 kitchen)
     (at tray104 kitchen)
     (at_kitchen_bread bread200)
     (at_kitchen_bread bread87)
     (at_kitchen_content content328)
     (at_kitchen_content content19)
     (no_gluten_bread bread200)
     (no_gluten_content content328)
     (allergic_gluten child112)
     (not_allergic_gluten child175)
     (waiting child112 table27)
     (waiting child175 table395)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child112)
     (served child175)
    )
  )
)
