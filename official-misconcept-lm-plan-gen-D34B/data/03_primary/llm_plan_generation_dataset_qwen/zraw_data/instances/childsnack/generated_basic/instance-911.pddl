; child-snack task with 3 children and 0.31 gluten factor 
; constant factor of 1.5
; random seed: 188106

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child25 child278 child201 - child
    bread312 bread106 bread457 - bread-portion
    content244 content453 content348 - content-portion
    tray191 - tray
    table454 table212 table340 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 - sandwich
  )
  (:init
     (at tray191 kitchen)
     (at_kitchen_bread bread312)
     (at_kitchen_bread bread106)
     (at_kitchen_bread bread457)
     (at_kitchen_content content244)
     (at_kitchen_content content453)
     (at_kitchen_content content348)
     (not_allergic_gluten child278)
     (not_allergic_gluten child201)
     (not_allergic_gluten child25)
     (waiting child25 table340)
     (waiting child278 table454)
     (waiting child201 table212)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
  )
  (:goal
    (and
     (served child25)
     (served child278)
     (served child201)
    )
  )
)
