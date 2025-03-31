; child-snack task with 3 children and 0.22 gluten factor 
; constant factor of 1.9
; random seed: 184515

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child251 child79 child176 - child
    bread28 bread442 bread386 - bread-portion
    content153 content229 content88 - content-portion
    tray116 - tray
    table434 table184 table279 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray116 kitchen)
     (at_kitchen_bread bread28)
     (at_kitchen_bread bread442)
     (at_kitchen_bread bread386)
     (at_kitchen_content content153)
     (at_kitchen_content content229)
     (at_kitchen_content content88)
     (not_allergic_gluten child176)
     (not_allergic_gluten child79)
     (not_allergic_gluten child251)
     (waiting child251 table184)
     (waiting child79 table184)
     (waiting child176 table184)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child251)
     (served child79)
     (served child176)
    )
  )
)
