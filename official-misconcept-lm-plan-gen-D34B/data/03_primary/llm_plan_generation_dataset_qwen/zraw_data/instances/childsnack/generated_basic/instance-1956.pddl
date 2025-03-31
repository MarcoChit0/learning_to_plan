; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.3
; random seed: 347437

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child190 child182 - child
    bread484 bread449 - bread-portion
    content398 content43 - content-portion
    tray325 - tray
    table45 table135 table34 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray325 kitchen)
     (at_kitchen_bread bread484)
     (at_kitchen_bread bread449)
     (at_kitchen_content content398)
     (at_kitchen_content content43)
     (no_gluten_bread bread449)
     (no_gluten_content content43)
     (allergic_gluten child190)
     (not_allergic_gluten child182)
     (waiting child190 table135)
     (waiting child182 table45)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child190)
     (served child182)
    )
  )
)
