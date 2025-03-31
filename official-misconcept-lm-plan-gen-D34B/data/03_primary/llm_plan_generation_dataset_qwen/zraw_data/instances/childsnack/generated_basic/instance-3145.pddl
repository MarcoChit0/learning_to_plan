; child-snack task with 2 children and 0.48 gluten factor 
; constant factor of 1.9
; random seed: 614058

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child457 child89 - child
    bread379 bread158 - bread-portion
    content223 content482 - content-portion
    tray80 tray228 - tray
    table443 table165 table487 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray80 kitchen)
     (at tray228 kitchen)
     (at_kitchen_bread bread379)
     (at_kitchen_bread bread158)
     (at_kitchen_content content223)
     (at_kitchen_content content482)
     (not_allergic_gluten child457)
     (not_allergic_gluten child89)
     (waiting child457 table443)
     (waiting child89 table165)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child457)
     (served child89)
    )
  )
)
