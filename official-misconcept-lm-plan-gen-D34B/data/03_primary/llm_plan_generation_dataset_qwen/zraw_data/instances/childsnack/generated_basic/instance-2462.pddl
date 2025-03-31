; child-snack task with 2 children and 0.39 gluten factor 
; constant factor of 1.9
; random seed: 74406

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child121 child310 - child
    bread224 bread151 - bread-portion
    content240 content246 - content-portion
    tray4 tray308 - tray
    table461 table463 table195 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray4 kitchen)
     (at tray308 kitchen)
     (at_kitchen_bread bread224)
     (at_kitchen_bread bread151)
     (at_kitchen_content content240)
     (at_kitchen_content content246)
     (not_allergic_gluten child121)
     (not_allergic_gluten child310)
     (waiting child121 table195)
     (waiting child310 table461)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child121)
     (served child310)
    )
  )
)
