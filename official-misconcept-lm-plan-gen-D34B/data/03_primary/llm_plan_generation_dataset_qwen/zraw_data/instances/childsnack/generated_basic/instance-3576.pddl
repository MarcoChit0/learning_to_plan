; child-snack task with 3 children and 0.36 gluten factor 
; constant factor of 1.3
; random seed: 653961

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child495 child434 child242 - child
    bread210 bread72 bread240 - bread-portion
    content323 content68 content37 - content-portion
    tray99 tray49 tray425 - tray
    table267 table151 table382 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray99 kitchen)
     (at tray49 kitchen)
     (at tray425 kitchen)
     (at_kitchen_bread bread210)
     (at_kitchen_bread bread72)
     (at_kitchen_bread bread240)
     (at_kitchen_content content323)
     (at_kitchen_content content68)
     (at_kitchen_content content37)
     (no_gluten_bread bread240)
     (no_gluten_content content68)
     (allergic_gluten child242)
     (not_allergic_gluten child495)
     (not_allergic_gluten child434)
     (waiting child495 table382)
     (waiting child434 table267)
     (waiting child242 table382)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child495)
     (served child434)
     (served child242)
    )
  )
)
