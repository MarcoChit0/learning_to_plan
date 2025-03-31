; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 912536

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child256 child415 - child
    bread32 bread116 - bread-portion
    content225 content347 - content-portion
    tray169 tray119 tray270 - tray
    table138 table44 table357 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray169 kitchen)
     (at tray119 kitchen)
     (at tray270 kitchen)
     (at_kitchen_bread bread32)
     (at_kitchen_bread bread116)
     (at_kitchen_content content225)
     (at_kitchen_content content347)
     (no_gluten_bread bread116)
     (no_gluten_content content225)
     (allergic_gluten child256)
     (not_allergic_gluten child415)
     (waiting child256 table357)
     (waiting child415 table44)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child256)
     (served child415)
    )
  )
)
