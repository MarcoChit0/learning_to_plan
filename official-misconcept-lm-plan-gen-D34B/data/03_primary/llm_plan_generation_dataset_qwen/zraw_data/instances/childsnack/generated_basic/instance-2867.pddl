; child-snack task with 1 children and 0.23 gluten factor 
; constant factor of 1.7
; random seed: 689759

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child262 - child
    bread326 - bread-portion
    content157 - content-portion
    tray44 tray17 - tray
    table231 table233 table287 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray44 kitchen)
     (at tray17 kitchen)
     (at_kitchen_bread bread326)
     (at_kitchen_content content157)
     (not_allergic_gluten child262)
     (waiting child262 table287)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child262)
    )
  )
)
