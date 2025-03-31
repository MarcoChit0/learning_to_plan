; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.7
; random seed: 392134

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child146 child52 - child
    bread317 bread255 - bread-portion
    content443 content370 - content-portion
    tray271 tray420 - tray
    table375 table254 table281 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray271 kitchen)
     (at tray420 kitchen)
     (at_kitchen_bread bread317)
     (at_kitchen_bread bread255)
     (at_kitchen_content content443)
     (at_kitchen_content content370)
     (no_gluten_bread bread255)
     (no_gluten_content content370)
     (allergic_gluten child52)
     (not_allergic_gluten child146)
     (waiting child146 table281)
     (waiting child52 table254)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child146)
     (served child52)
    )
  )
)
