; child-snack task with 2 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 837958

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child170 child251 - child
    bread93 bread166 - bread-portion
    content269 content266 - content-portion
    tray86 tray241 tray165 tray139 - tray
    table472 table451 table283 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray86 kitchen)
     (at tray241 kitchen)
     (at tray165 kitchen)
     (at tray139 kitchen)
     (at_kitchen_bread bread93)
     (at_kitchen_bread bread166)
     (at_kitchen_content content269)
     (at_kitchen_content content266)
     (not_allergic_gluten child170)
     (not_allergic_gluten child251)
     (waiting child170 table451)
     (waiting child251 table283)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child170)
     (served child251)
    )
  )
)
