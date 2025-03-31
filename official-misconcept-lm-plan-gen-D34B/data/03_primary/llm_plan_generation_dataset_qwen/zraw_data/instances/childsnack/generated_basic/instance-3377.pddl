; child-snack task with 3 children and 0.34 gluten factor 
; constant factor of 1.5
; random seed: 44368

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child314 child310 child23 - child
    bread151 bread246 bread72 - bread-portion
    content140 content62 content242 - content-portion
    tray44 tray25 tray373 - tray
    table242 table68 table431 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 - sandwich
  )
  (:init
     (at tray44 kitchen)
     (at tray25 kitchen)
     (at tray373 kitchen)
     (at_kitchen_bread bread151)
     (at_kitchen_bread bread246)
     (at_kitchen_bread bread72)
     (at_kitchen_content content140)
     (at_kitchen_content content62)
     (at_kitchen_content content242)
     (no_gluten_bread bread246)
     (no_gluten_content content242)
     (allergic_gluten child23)
     (not_allergic_gluten child314)
     (not_allergic_gluten child310)
     (waiting child314 table68)
     (waiting child310 table242)
     (waiting child23 table68)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
  )
  (:goal
    (and
     (served child314)
     (served child310)
     (served child23)
    )
  )
)
