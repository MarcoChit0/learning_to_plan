; child-snack task with 1 children and 0.2 gluten factor 
; constant factor of 1.9
; random seed: 792568

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child392 - child
    bread174 - bread-portion
    content496 - content-portion
    tray137 tray20 tray237 tray203 - tray
    table71 table447 table444 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray137 kitchen)
     (at tray20 kitchen)
     (at tray237 kitchen)
     (at tray203 kitchen)
     (at_kitchen_bread bread174)
     (at_kitchen_content content496)
     (not_allergic_gluten child392)
     (waiting child392 table444)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child392)
    )
  )
)
