; child-snack task with 3 children and 0.4 gluten factor 
; constant factor of 1.5
; random seed: 35194

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child289 child29 child161 - child
    bread68 bread266 bread70 - bread-portion
    content66 content288 content252 - content-portion
    tray308 tray72 tray410 tray446 - tray
    table493 table31 table148 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 - sandwich
  )
  (:init
     (at tray308 kitchen)
     (at tray72 kitchen)
     (at tray410 kitchen)
     (at tray446 kitchen)
     (at_kitchen_bread bread68)
     (at_kitchen_bread bread266)
     (at_kitchen_bread bread70)
     (at_kitchen_content content66)
     (at_kitchen_content content288)
     (at_kitchen_content content252)
     (no_gluten_bread bread68)
     (no_gluten_content content66)
     (allergic_gluten child29)
     (not_allergic_gluten child161)
     (not_allergic_gluten child289)
     (waiting child289 table493)
     (waiting child29 table31)
     (waiting child161 table148)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
  )
  (:goal
    (and
     (served child289)
     (served child29)
     (served child161)
    )
  )
)
