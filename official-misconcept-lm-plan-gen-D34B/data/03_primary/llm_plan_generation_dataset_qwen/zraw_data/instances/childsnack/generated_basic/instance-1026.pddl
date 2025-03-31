; child-snack task with 3 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 490291

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child102 child184 child28 - child
    bread35 bread322 bread338 - bread-portion
    content251 content288 content107 - content-portion
    tray15 - tray
    table275 table224 table272 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray15 kitchen)
     (at_kitchen_bread bread35)
     (at_kitchen_bread bread322)
     (at_kitchen_bread bread338)
     (at_kitchen_content content251)
     (at_kitchen_content content288)
     (at_kitchen_content content107)
     (not_allergic_gluten child102)
     (not_allergic_gluten child184)
     (not_allergic_gluten child28)
     (waiting child102 table224)
     (waiting child184 table224)
     (waiting child28 table224)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child102)
     (served child184)
     (served child28)
    )
  )
)
