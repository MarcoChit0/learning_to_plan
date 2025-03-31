; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.9
; random seed: 736848

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child24 child26 - child
    bread224 bread430 - bread-portion
    content290 content359 - content-portion
    tray280 tray409 tray86 tray198 - tray
    table278 table90 table477 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray280 kitchen)
     (at tray409 kitchen)
     (at tray86 kitchen)
     (at tray198 kitchen)
     (at_kitchen_bread bread224)
     (at_kitchen_bread bread430)
     (at_kitchen_content content290)
     (at_kitchen_content content359)
     (not_allergic_gluten child24)
     (not_allergic_gluten child26)
     (waiting child24 table90)
     (waiting child26 table477)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child24)
     (served child26)
    )
  )
)
