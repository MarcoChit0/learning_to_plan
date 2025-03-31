; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.9
; random seed: 20489

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child62 child122 - child
    bread10 bread414 - bread-portion
    content147 content427 - content-portion
    tray413 tray326 tray205 tray278 - tray
    table324 table176 table421 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray413 kitchen)
     (at tray326 kitchen)
     (at tray205 kitchen)
     (at tray278 kitchen)
     (at_kitchen_bread bread10)
     (at_kitchen_bread bread414)
     (at_kitchen_content content147)
     (at_kitchen_content content427)
     (not_allergic_gluten child122)
     (not_allergic_gluten child62)
     (waiting child62 table176)
     (waiting child122 table324)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child62)
     (served child122)
    )
  )
)
