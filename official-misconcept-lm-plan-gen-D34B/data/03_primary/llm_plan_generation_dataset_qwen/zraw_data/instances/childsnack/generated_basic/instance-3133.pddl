; child-snack task with 2 children and 0.67 gluten factor 
; constant factor of 1.5
; random seed: 942340

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child80 child289 - child
    bread104 bread159 - bread-portion
    content169 content483 - content-portion
    tray73 - tray
    table281 table418 table156 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray73 kitchen)
     (at_kitchen_bread bread104)
     (at_kitchen_bread bread159)
     (at_kitchen_content content169)
     (at_kitchen_content content483)
     (no_gluten_bread bread159)
     (no_gluten_content content169)
     (allergic_gluten child80)
     (not_allergic_gluten child289)
     (waiting child80 table418)
     (waiting child289 table281)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child80)
     (served child289)
    )
  )
)
