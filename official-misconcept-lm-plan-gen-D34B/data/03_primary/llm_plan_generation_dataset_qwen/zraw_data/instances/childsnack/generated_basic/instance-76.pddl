; child-snack task with 2 children and 0.36 gluten factor 
; constant factor of 1.9
; random seed: 134374

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child217 child276 - child
    bread371 bread44 - bread-portion
    content20 content97 - content-portion
    tray299 tray410 - tray
    table372 table187 table480 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray299 kitchen)
     (at tray410 kitchen)
     (at_kitchen_bread bread371)
     (at_kitchen_bread bread44)
     (at_kitchen_content content20)
     (at_kitchen_content content97)
     (not_allergic_gluten child217)
     (not_allergic_gluten child276)
     (waiting child217 table372)
     (waiting child276 table187)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child217)
     (served child276)
    )
  )
)
