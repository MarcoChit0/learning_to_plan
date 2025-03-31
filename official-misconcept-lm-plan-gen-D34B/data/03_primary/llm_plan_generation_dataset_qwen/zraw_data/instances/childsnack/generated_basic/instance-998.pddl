; child-snack task with 3 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 434045

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child342 child251 child304 - child
    bread453 bread475 bread99 - bread-portion
    content165 content277 content412 - content-portion
    tray114 - tray
    table176 table0 table62 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray114 kitchen)
     (at_kitchen_bread bread453)
     (at_kitchen_bread bread475)
     (at_kitchen_bread bread99)
     (at_kitchen_content content165)
     (at_kitchen_content content277)
     (at_kitchen_content content412)
     (not_allergic_gluten child304)
     (not_allergic_gluten child342)
     (not_allergic_gluten child251)
     (waiting child342 table0)
     (waiting child251 table62)
     (waiting child304 table176)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child342)
     (served child251)
     (served child304)
    )
  )
)
