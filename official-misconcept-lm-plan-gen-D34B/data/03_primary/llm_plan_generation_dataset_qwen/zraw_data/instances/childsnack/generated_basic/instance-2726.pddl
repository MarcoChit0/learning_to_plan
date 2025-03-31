; child-snack task with 3 children and 0.23 gluten factor 
; constant factor of 1.9
; random seed: 773856

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child265 child153 child466 - child
    bread440 bread152 bread79 - bread-portion
    content288 content188 content183 - content-portion
    tray410 tray159 tray274 - tray
    table477 table432 table226 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray410 kitchen)
     (at tray159 kitchen)
     (at tray274 kitchen)
     (at_kitchen_bread bread440)
     (at_kitchen_bread bread152)
     (at_kitchen_bread bread79)
     (at_kitchen_content content288)
     (at_kitchen_content content188)
     (at_kitchen_content content183)
     (not_allergic_gluten child466)
     (not_allergic_gluten child153)
     (not_allergic_gluten child265)
     (waiting child265 table477)
     (waiting child153 table432)
     (waiting child466 table226)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child265)
     (served child153)
     (served child466)
    )
  )
)
