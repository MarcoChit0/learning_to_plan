; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 287821

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child204 child483 - child
    bread277 bread224 - bread-portion
    content202 content265 - content-portion
    tray51 tray150 tray262 tray214 - tray
    table346 table293 table75 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray51 kitchen)
     (at tray150 kitchen)
     (at tray262 kitchen)
     (at tray214 kitchen)
     (at_kitchen_bread bread277)
     (at_kitchen_bread bread224)
     (at_kitchen_content content202)
     (at_kitchen_content content265)
     (no_gluten_bread bread224)
     (no_gluten_content content202)
     (allergic_gluten child483)
     (not_allergic_gluten child204)
     (waiting child204 table346)
     (waiting child483 table293)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child204)
     (served child483)
    )
  )
)
