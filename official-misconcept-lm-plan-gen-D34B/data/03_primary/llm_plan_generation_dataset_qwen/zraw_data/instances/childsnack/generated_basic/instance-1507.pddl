; child-snack task with 2 children and 0.66 gluten factor 
; constant factor of 1.3
; random seed: 440278

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child273 child316 - child
    bread478 bread369 - bread-portion
    content249 content103 - content-portion
    tray278 tray111 tray276 - tray
    table285 table200 table278 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray278 kitchen)
     (at tray111 kitchen)
     (at tray276 kitchen)
     (at_kitchen_bread bread478)
     (at_kitchen_bread bread369)
     (at_kitchen_content content249)
     (at_kitchen_content content103)
     (no_gluten_bread bread369)
     (no_gluten_content content103)
     (allergic_gluten child273)
     (not_allergic_gluten child316)
     (waiting child273 table285)
     (waiting child316 table285)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child273)
     (served child316)
    )
  )
)
