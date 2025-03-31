; child-snack task with 2 children and 0.48 gluten factor 
; constant factor of 1.9
; random seed: 48136

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child11 child182 - child
    bread45 bread182 - bread-portion
    content63 content414 - content-portion
    tray405 tray173 - tray
    table55 table64 table116 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray405 kitchen)
     (at tray173 kitchen)
     (at_kitchen_bread bread45)
     (at_kitchen_bread bread182)
     (at_kitchen_content content63)
     (at_kitchen_content content414)
     (not_allergic_gluten child182)
     (not_allergic_gluten child11)
     (waiting child11 table55)
     (waiting child182 table64)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child11)
     (served child182)
    )
  )
)
