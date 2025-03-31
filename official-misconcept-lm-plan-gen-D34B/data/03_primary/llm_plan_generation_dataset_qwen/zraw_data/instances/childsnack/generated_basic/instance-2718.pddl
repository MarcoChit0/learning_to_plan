; child-snack task with 3 children and 0.23 gluten factor 
; constant factor of 1.9
; random seed: 79064

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child198 child303 child63 - child
    bread40 bread58 bread147 - bread-portion
    content435 content184 content319 - content-portion
    tray23 tray454 tray255 - tray
    table281 table117 table123 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray23 kitchen)
     (at tray454 kitchen)
     (at tray255 kitchen)
     (at_kitchen_bread bread40)
     (at_kitchen_bread bread58)
     (at_kitchen_bread bread147)
     (at_kitchen_content content435)
     (at_kitchen_content content184)
     (at_kitchen_content content319)
     (not_allergic_gluten child63)
     (not_allergic_gluten child198)
     (not_allergic_gluten child303)
     (waiting child198 table123)
     (waiting child303 table123)
     (waiting child63 table123)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child198)
     (served child303)
     (served child63)
    )
  )
)
