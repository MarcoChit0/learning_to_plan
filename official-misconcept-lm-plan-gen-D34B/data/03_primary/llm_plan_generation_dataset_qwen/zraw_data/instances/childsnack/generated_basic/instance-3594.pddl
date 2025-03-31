; child-snack task with 3 children and 0.36 gluten factor 
; constant factor of 1.3
; random seed: 76014

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child110 child193 child449 - child
    bread78 bread91 bread284 - bread-portion
    content294 content57 content482 - content-portion
    tray146 tray40 tray83 - tray
    table218 table185 table454 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray146 kitchen)
     (at tray40 kitchen)
     (at tray83 kitchen)
     (at_kitchen_bread bread78)
     (at_kitchen_bread bread91)
     (at_kitchen_bread bread284)
     (at_kitchen_content content294)
     (at_kitchen_content content57)
     (at_kitchen_content content482)
     (no_gluten_bread bread284)
     (no_gluten_content content482)
     (allergic_gluten child110)
     (not_allergic_gluten child449)
     (not_allergic_gluten child193)
     (waiting child110 table185)
     (waiting child193 table454)
     (waiting child449 table454)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child110)
     (served child193)
     (served child449)
    )
  )
)
