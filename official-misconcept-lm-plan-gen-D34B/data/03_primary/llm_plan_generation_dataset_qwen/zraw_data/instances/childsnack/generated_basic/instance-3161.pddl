; child-snack task with 2 children and 0.48 gluten factor 
; constant factor of 1.9
; random seed: 268180

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child12 child393 - child
    bread254 bread147 - bread-portion
    content41 content265 - content-portion
    tray166 tray94 - tray
    table425 table174 table92 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray166 kitchen)
     (at tray94 kitchen)
     (at_kitchen_bread bread254)
     (at_kitchen_bread bread147)
     (at_kitchen_content content41)
     (at_kitchen_content content265)
     (not_allergic_gluten child12)
     (not_allergic_gluten child393)
     (waiting child12 table174)
     (waiting child393 table92)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child12)
     (served child393)
    )
  )
)
